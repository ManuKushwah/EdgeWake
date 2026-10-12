#include "health.h"

#include <stdatomic.h>
#include <stdio.h>

#include "app_config.h"
#include "esp_app_desc.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "ota.h"
#include "telemetry.h"
#include "wifi.h"

static const char *TAG = "health";

#define CPU_WINDOW_S        5
#define STALL_REBOOT_S      10

static StaticTask_t s_tcb;
static StackType_t s_stack[EW_STACK_HEALTH];
static _Atomic uint32_t s_cpu_permille;  /* written by the monitor, read by snapshots */

static const char *reset_reason_str(esp_reset_reason_t r)
{
    switch (r) {
    case ESP_RST_POWERON:  return "power_on";
    case ESP_RST_SW:       return "software";
    case ESP_RST_PANIC:    return "panic";
    case ESP_RST_INT_WDT:  return "int_wdt";
    case ESP_RST_TASK_WDT: return "task_wdt";
    case ESP_RST_WDT:      return "other_wdt";
    case ESP_RST_BROWNOUT: return "brownout";
    case ESP_RST_DEEPSLEEP:return "deep_sleep";
    default:               return "other";
    }
}

static void monitor_task(void *arg)
{
    (void)arg;
    ESP_ERROR_CHECK(esp_task_wdt_add(NULL));

    kws_stats_t k;
    audio_stats_t a;
    kws_get_stats(&k);
    audio_get_stats(&a);
    uint64_t last_busy = k.busy_us;
    int64_t last_t = esp_timer_get_time();
    uint32_t last_chunks = a.chunks, last_hops = k.engine.hops;
    int stalled_s = 0, tick = 0;

    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        esp_task_wdt_reset();
        kws_get_stats(&k);
        audio_get_stats(&a);

        /* Software liveness check on top of the TWDT: a task can be alive
         * (feeding its watchdog) while the data path is stuck, e.g. the I2S
         * peripheral stops delivering or the KWS task stops draining. */
        const bool audio_moving = a.chunks != last_chunks;
        const bool kws_moving = k.engine.hops != last_hops;
        stalled_s = (audio_moving && kws_moving) ? 0 : stalled_s + 1;
        last_chunks = a.chunks;
        last_hops = k.engine.hops;
        if (stalled_s == 3) {
            ESP_LOGW(TAG, "pipeline stalled (audio %s, kws %s)", audio_moving ? "ok" : "STUCK",
                     kws_moving ? "ok" : "STUCK");
        }
        if (stalled_s >= STALL_REBOOT_S) {
            ESP_LOGE(TAG, "pipeline stalled for %d s: rebooting", stalled_s);
            esp_restart();
        }

        /* KWS CPU load = busy time / wall time over the window. */
        if (++tick % CPU_WINDOW_S == 0) {
            const int64_t now = esp_timer_get_time();
            const uint64_t busy = k.busy_us - last_busy;
            atomic_store(&s_cpu_permille, (uint32_t)(busy * 1000 / (uint64_t)(now - last_t)));
            last_busy = k.busy_us;
            last_t = now;
        }
    }
}

esp_err_t health_start(void)
{
    ESP_LOGI(TAG, "last reset: %s", reset_reason_str(esp_reset_reason()));
    TaskHandle_t h = xTaskCreateStaticPinnedToCore(monitor_task, "health", EW_STACK_HEALTH, NULL,
                                                   EW_PRIO_HEALTH, s_stack, &s_tcb, EW_CORE_NET);
    return h ? ESP_OK : ESP_FAIL;
}

void health_snapshot(health_t *h)
{
    h->uptime_s = (uint32_t)(esp_timer_get_time() / 1000000);
    h->heap_free = esp_get_free_heap_size();
    h->heap_min_free = esp_get_minimum_free_heap_size();
    audio_get_stats(&h->audio);
    kws_get_stats(&h->kws);
    h->kws_cpu_pct = (float)atomic_load(&s_cpu_permille) / 10.0f;
    h->vad_skip_pct = h->kws.engine.infer_slots
                          ? 100.0f * (float)h->kws.engine.skipped_by_vad / (float)h->kws.engine.infer_slots
                          : 0.0f;
    h->infer_us_avg = h->kws.infer_count ? (uint32_t)(h->kws.infer_us_sum / h->kws.infer_count) : 0;
    /* ESP-IDF stacks are measured in bytes. */
    h->stack_free_audio = audio_task_handle() ? uxTaskGetStackHighWaterMark(audio_task_handle()) : 0;
    h->stack_free_kws = kws_task_handle() ? uxTaskGetStackHighWaterMark(kws_task_handle()) : 0;
    h->rssi = wifi_rssi();
    h->mqtt_dropped = telemetry_dropped();
    h->reset_reason = reset_reason_str(esp_reset_reason());
    h->fw_version = esp_app_get_description()->version;
    h->ota_state = ota_state_str();
    h->partition = ota_running_partition();
}

int health_to_json(const health_t *h, char *buf, size_t len)
{
    return snprintf(buf, len,
        "{\"fw\":\"%s\",\"part\":\"%s\",\"ota\":\"%s\",\"reset\":\"%s\",\"up_s\":%lu,"
        "\"heap\":%lu,\"heap_min\":%lu,\"rssi\":%d,"
        "\"audio\":{\"chunks\":%lu,\"read_err\":%lu,\"dma_overflows\":%lu,\"dropped\":%lu,\"peak\":%ld},"
        "\"kws\":{\"hops\":%lu,\"slots\":%lu,\"inferences\":%lu,\"vad_skipped\":%lu,\"overload_skipped\":%lu,"
        "\"vad_skip_pct\":%.1f,\"detections\":%lu,\"errors\":%lu,\"vad\":%s,\"thr\":%.2f},"
        "\"timing\":{\"cpu_pct\":%.1f,\"infer_us_avg\":%lu,\"infer_us_max\":%lu,\"hop_us_max\":%lu},"
        "\"stack_free\":{\"audio\":%lu,\"kws\":%lu},\"events_dropped\":%lu,\"mqtt_dropped\":%lu}",
        h->fw_version, h->partition, h->ota_state, h->reset_reason, (unsigned long)h->uptime_s,
        (unsigned long)h->heap_free, (unsigned long)h->heap_min_free, h->rssi,
        (unsigned long)h->audio.chunks, (unsigned long)h->audio.read_errors,
        (unsigned long)h->audio.dma_overflows,
        (unsigned long)h->audio.dropped_samples, (long)h->audio.last_peak,
        (unsigned long)h->kws.engine.hops, (unsigned long)h->kws.engine.infer_slots,
        (unsigned long)h->kws.engine.inferences, (unsigned long)h->kws.engine.skipped_by_vad,
        (unsigned long)h->kws.engine.skipped_overload,
        (double)h->vad_skip_pct, (unsigned long)h->kws.engine.detections,
        (unsigned long)h->kws.engine.infer_errors, h->kws.vad_enabled ? "true" : "false",
        (double)h->kws.threshold, (double)h->kws_cpu_pct, (unsigned long)h->infer_us_avg,
        (unsigned long)h->kws.infer_us_max, (unsigned long)h->kws.hop_us_max,
        (unsigned long)h->stack_free_audio, (unsigned long)h->stack_free_kws,
        (unsigned long)h->kws.events_dropped, (unsigned long)h->mqtt_dropped);
}
