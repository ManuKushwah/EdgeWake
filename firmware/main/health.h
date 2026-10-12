/*
 * health.h - device health: CPU load of the KWS pipeline, heap, stack
 * high-water marks, audio/KWS counters, reset reason, and a liveness monitor
 * that reboots the device if the audio pipeline stalls.
 */
#ifndef HEALTH_H
#define HEALTH_H

#include <stddef.h>
#include <stdint.h>

#include "audio_capture.h"
#include "esp_err.h"
#include "kws_task.h"

typedef struct {
    uint32_t uptime_s;
    uint32_t heap_free;
    uint32_t heap_min_free;
    audio_stats_t audio;
    kws_stats_t kws;
    float kws_cpu_pct;          /* share of core 1 used by the KWS task (last window) */
    float vad_skip_pct;         /* inference slots skipped by the VAD gate (cumulative) */
    uint32_t infer_us_avg;
    uint32_t stack_free_audio;  /* bytes never touched (high-water mark) */
    uint32_t stack_free_kws;
    int rssi;
    const char *reset_reason;
    const char *fw_version;
    const char *ota_state;
    const char *partition;
} health_t;

esp_err_t health_start(void);
void health_snapshot(health_t *h);

/* Compact JSON for MQTT; returns the length written (snprintf semantics). */
int health_to_json(const health_t *h, char *buf, size_t len);

#endif /* HEALTH_H */
